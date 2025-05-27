import torch
from models.autoVC.autovc import AutoVC

# Create dummy inputs
batch_size = 4
time_steps = 128
mel_dim = 80
num_emotions = 5

source_mel = torch.randn(batch_size, time_steps, mel_dim)  # (B, T, 80)
target_mel = torch.randn(batch_size, time_steps, mel_dim)  # (B, T', 80)
emotion_label = torch.randint(0, num_emotions, (batch_size,))  # (B,)

# Initialize model
model = AutoVC(
    content_dim=mel_dim,
    speaker_dim=mel_dim,
    content_emb_dim=128,
    speaker_emb_dim=128,
    emotion_emb_dim=128,
    num_emotions=num_emotions,
    bottleneck_dim=384,
    mel_dim=mel_dim
)

# Set to eval mode (not training here)
model.eval()

# Forward pass
with torch.no_grad():
    mel_output = model(source_mel, target_mel, emotion_label)

# Print output shape
print("Output shape:", mel_output.shape)
assert mel_output.shape == (batch_size, time_steps, mel_dim)
print("Forward pass successful")
