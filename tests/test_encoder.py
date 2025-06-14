import torch

from models.autoVC.content_encoder import ContentEncoder

mel = torch.randn(1, 100, 80)  # Simulate (B=1, T=100, mel=80)
encoder = ContentEncoder()
with torch.no_grad():
    out = encoder(mel)
print(out.shape)  # should be (1, 100, 128)
print("Stats:", out.min().item(), out.max().item(), out.mean().item(), out.std().item())
