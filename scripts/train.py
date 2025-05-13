from mel_dataset import MelDataset
from torch.utils.data import DataLoader

# DataSet LOading
dataset = MelDataset(config['dataset_path'], config['emotion_map'], ...)
loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=...)

# Train loop
for epoch in range(num_epochs):
    for mel, speaker_id, emotion_label in loader:
