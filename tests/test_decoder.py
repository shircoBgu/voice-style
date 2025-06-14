import torch

from models.autoVC.decoder import Decoder

dummy_input = torch.randn(1, 100, 384)
decoder = Decoder()
out = decoder(dummy_input)
print(out.shape)  # should be (1, 100, 80)
print(out.mean(), out.std())
