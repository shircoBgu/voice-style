import os
import glob
import json
import torch
from pathlib import Path
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from hifigan.models import Generator
from hifigan.env import AttrDict


# Helper function to extract epoch number from checkpoint filename
def extract_epoch_num(filename):
    try:
        return int(filename.split("epoch")[-1].split(".")[0])
    except:
        return -1


class VoiceConverter:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.emo2idx = None
        self.speaker2idx = None
        self.autovc_model = None
        self.hifigan_model = None

    # Load AutoVC model from latest checkpoint
    def load_autovc(self, model_class, checkpoint_path=None):
        if checkpoint_path is None:
            ckpt_dir = self.config["training"].get("checkpoint_dir", "checkpoints")
            candidates = glob.glob(os.path.join(ckpt_dir, "checkpoint_epoch*.pt"))
            if not candidates:
                raise FileNotFoundError("No AutoVC checkpoint found")
            candidates = sorted(candidates, key=extract_epoch_num)
            checkpoint_path = candidates[-1]

        print(f"Loading AutoVC from {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(state, dict) and "model_state" in state:
            # Initialize model with correct emotion/speaker dimensions
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

    # Load Universal HiFi-GAN vocoder
    def load_hifigan(self):
        hifigan_dir = self.config["paths"]["hifigan_pretrained"]
        config_file = os.path.join(hifigan_dir, "config.json")
        checkpoint_file = os.path.join(hifigan_dir, "generator_v1")

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

    # Load alternative WaveNet vocoder
    def load_wavenet(self):
        from models.wavenet import WaveNetVocoderWrapper
        wavenet_dir = self.config["paths"]["pretrained_wavenet"]
        checkpoint_path = os.path.join(wavenet_dir, "checkpoint_step000740000_ema.pth")
        hparams_path = os.path.join(wavenet_dir, "hparams.json")

        if not os.path.exists(checkpoint_path) or not os.path.exists(hparams_path):
            raise FileNotFoundError("Missing WaveNet checkpoint or config")

        self.wavenet_model = WaveNetVocoderWrapper(checkpoint_path, hparams_path, self.device)

    # Load audio file and convert to HiFi-GAN-style mel-spectrogram
    def load_audio_as_mel(self, path, target_len=None):
        waveform, sr = torchaudio.load(path)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        target_sr = self.config['dataset']['sample_rate']
        if sr != target_sr:
            waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)

        # Normalize amplitude to avoid clipping
        waveform = waveform / waveform.abs().max() * 0.95

        # MelSpectrogram parameters must match HiFi-GAN training config
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sr,
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            n_mels=80,
            f_min=0,
            f_max=8000,
            power=1.0,
            center=False,
            window_fn=torch.hann_window
        )
        mel = mel_transform(waveform)
        mel = torch.log(mel + 1e-9).squeeze(0).transpose(0, 1)  # (T, 80)

        # Optional padding or trimming
        if target_len:
            mel = mel[:target_len] if mel.size(0) > target_len else torch.cat(
                [mel, mel[-1:].repeat(target_len - mel.size(0), 1)]
            )

        return mel.unsqueeze(0).to(self.device)  # (1, T, 80)

    # Load mel directly from saved .npy file
    def load_mel_from_npy(self, path, target_len=None):
        """
        Load a precomputed mel spectrogram saved as (80, T) .npy file and prepare it for inference.
        Returns: Tensor of shape (1, T, 80)
        """
        mel = np.load(path)  # (80, T)
        mel = torch.tensor(mel.T, dtype=torch.float32)  # (T, 80)

        if target_len:
            mel = mel[:target_len] if mel.size(0) > target_len else torch.cat(
                [mel, mel[-1:].repeat(target_len - mel.size(0), 1)]
            )

        return mel.unsqueeze(0).to(self.device)

    # Run inference: convert source to target style with emotion, synthesize with vocoder
    def convert(self, source_path, target_path, emotion_label, output_path, use_npy=False):
        if self.autovc_model is None:
            raise RuntimeError("AutoVC model not loaded")
        if not (self.hifigan_model or hasattr(self, 'wavenet_model')):
            raise RuntimeError("No vocoder loaded")

        # Load mel spectrograms from audio or precomputed npy
        load_mel = self.load_mel_from_npy if use_npy else self.load_audio_as_mel
        source_mel = load_mel(source_path)
        target_mel = load_mel(target_path, target_len=source_mel.shape[1])
        emotion_tensor = torch.tensor([emotion_label], dtype=torch.long).to(self.device)

        with torch.no_grad():
            mel_out,_,_,_,_,_ = self.autovc_model(source_mel, target_mel, emotion_tensor)
            mel_input = mel_out.transpose(1, 2)  # (B, 80, T)

            # Ensure mel dimensions match HiFi-GAN expectation
            if mel_input.shape[1] != 80:
                raise ValueError(f"Expected 80 mel channels for HiFi-GAN, got {mel_input.shape[1]}")

            # Generate waveform using selected vocoder
            if self.hifigan_model:
                audio = self.hifigan_model(mel_input).squeeze().cpu().numpy()
            elif self.wavenet_model:
                audio = self.wavenet_model(mel_out).squeeze().cpu().numpy()

            # Plot mel output from AutoVC
            mel_out_np = mel_out.squeeze(0).cpu().numpy().T  # (80, T)
            plt.figure(figsize=(10, 4))
            plt.imshow(mel_out_np, aspect='auto', origin='lower')
            plt.title("Mel-Spectrogram Output from AutoVC")
            plt.colorbar()
            plt.tight_layout()
            plot_path = os.path.join(os.path.dirname(output_path), "autovc_output_mel.png")
            plt.savefig(plot_path)
            print(f"Saved AutoVC mel-spectrogram to {plot_path}")

            # Normalize audio to int16 for WAV file saving
            audio = audio / np.max(np.abs(audio))
            audio = np.int16(audio * 32767)
            write(output_path, self.config["dataset"]["sample_rate"], audio)
            print(f"Audio saved to {output_path}")

            # Log stats
            print("AutoVC Output Mel Stats:")
            print("  Min:", mel_out.min().item())
            print("  Max:", mel_out.max().item())
            print("  Mean:", mel_out.mean().item())
            print("  Std:", mel_out.std().item())
            print("Emotion tensor used:", emotion_tensor)
