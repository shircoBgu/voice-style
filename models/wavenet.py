import json

import torch

from wavenet_vocoder.train import build_model
from wavenet_vocoder.hparams import hparams as default_hparams

class WaveNetVocoderWrapper:
    def __init__(self, checkpoint_path, hparams_path, device):
        self.device = device

        # Load raw JSON
        with open(hparams_path) as f:
            raw_dict = json.load(f)

        # Only include keys supported by the default hparams object
        valid_keys = default_hparams.values().keys()
        filtered = {k: v for k, v in raw_dict.items() if k in valid_keys}

        # Parse filtered hparams
        hparams = default_hparams.parse_json(json.dumps(filtered))

        # Build and load model
        from wavenet_vocoder.train import build_model
        self.model = build_model().to(device)

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

    def __call__(self, mel_tensor):
        mel_tensor = mel_tensor.transpose(1, 2).to(self.device)  # (1, 80, T)
        with torch.no_grad():
            audio = self.model.generate(mel_tensor, fast=True, tqdm=lambda x: x)
        return audio.squeeze().cpu().numpy()