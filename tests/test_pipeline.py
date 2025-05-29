import torch
from models.autoVC.autovc import AutoVC
from models.emotion_classifier import EmotionClassifier
from hifigan.models import Generator
from hifigan.env import AttrDict
import json
import os
print("Current working directory:", os.getcwd())

# ==== Config ====
num_emotions = 5
batch_size = 4
time_steps = 100
mel_dim = 80
hifigan_ckpt_dir = os.path.join("..", "models", "hifigan_pretrained")  # path to config.json + generator weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Load HiFi-GAN ====
def load_hifigan_model(path):
    config_path = os.path.join(path, "config.json")
    with open(config_path) as f:
        config = AttrDict(json.load(f))
    model = Generator(config).to(device)
    checkpoint_path = os.path.join(path, "generator_v1")
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state['generator'] if 'generator' in state else state)
    model.eval()
    model.remove_weight_norm()
    return model

hifigan = load_hifigan_model(hifigan_ckpt_dir)

# ==== Dummy Input ====
source_mel = torch.randn(batch_size, time_steps, mel_dim).to(device)
target_mel = torch.randn(batch_size, time_steps, mel_dim).to(device)
emotion_labels = torch.randint(0, num_emotions, (batch_size,), dtype=torch.long).to(device)

# ==== Load Models ====
autovc = AutoVC(num_emotions=num_emotions).to(device)
emotion_cls = EmotionClassifier(num_classes=num_emotions).to(device)

# ==== Forward ====
mel_pred = autovc(source_mel, target_mel, emotion_labels)
print("✅ AutoVC output shape:", mel_pred.shape)  # (B, T, 80)

# ==== Emotion Classification Loss ====
logits = emotion_cls(mel_pred)
loss_fn = torch.nn.CrossEntropyLoss()
loss = loss_fn(logits, emotion_labels)
print("✅ Emotion loss:", loss.item())

# ==== HiFi-GAN Inference ====
with torch.no_grad():
    mel_input = mel_pred.transpose(1, 2)  # HiFi-GAN expects shape (B, 80, T)
    audio_out = hifigan(mel_input)  # (B, audio_len)
    print("✅ HiFi-GAN audio output shape:", audio_out.shape)
