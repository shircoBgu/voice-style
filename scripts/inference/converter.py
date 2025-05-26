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
        self.emotion_map = {
            0: "neutral", 1: "happy", 2: "sad", 3: "angry", 4: "surprised"
        }

        self.autovc_model = None
        self.hifigan_model = None

    def load_autovc(self, model_class, checkpoint_path=None):
        model = model_class().to(self.device)

        if checkpoint_path is None:
            ckpt_dir = self.config["training"].get("checkpoint_dir", "checkpoints")
            candidates = glob.glob(os.path.join(ckpt_dir, "*.pt"))
            if not candidates:
                raise FileNotFoundError("No AutoVC checkpoint found")
            checkpoint_path = sorted(candidates)[-1]

        print(f"Loading AutoVC from {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(state if isinstance(state, dict) else state['model_state_dict'])
        model.eval()
        self.autovc_model = model

    def load_hifigan(self):
        hifigan_path = self.config["paths"]["pretrained_hifigan"]
        config_file = os.path.join(hifigan_path, "config.json")
        with open(config_file) as f:
            config = AttrDict(json.load(f))

        model = Generator(config).to(self.device)

        checkpoint_path = os.path.join(hifigan_path, "generator_v1")
        ckpt = torch.load(checkpoint_path, map_location=self.device)
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

    def convert(self, source_path, target_path, emotion_label, output_path):
        if not all([self.autovc_model, self.hifigan_model]):
            raise RuntimeError("Models not loaded")

        source_mel = self.load_audio_as_mel(source_path)
        target_mel = self.load_audio_as_mel(target_path, target_len=source_mel.shape[1])
        emotion_tensor = torch.tensor([emotion_label], dtype=torch.long).to(self.device)

        with torch.no_grad():
            mel_out = self.autovc_model(source_mel, target_mel, emotion_tensor)
            audio = self.hifigan_model(mel_out.transpose(1, 2)).squeeze().cpu().numpy()
            audio = audio / np.max(np.abs(audio))
            audio = np.int16(audio * 32767)

            write(output_path, self.config["dataset"]["sample_rate"], audio)
            print(f"Audio saved to {output_path}")