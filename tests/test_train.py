import torch
from torch.utils.data import DataLoader, TensorDataset
from models.autoVC.autovc import AutoVC
from models.EmotionClassifier import EmotionClassifier
from scripts.train import train

# Create dummy data
B, T, D = 4, 100, 80  # Batch, Time, Mel-dim
X = torch.randn(B, T, D)
Y = torch.randn(B, T, D)
labels = torch.randint(0, 5, (B,))

# Wrap into DataLoader
dataset = TensorDataset(X, Y, labels)
dataloader = DataLoader(dataset, batch_size=2)

# Models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoVC(num_emotions=5).to(device)
emotion_classifier = EmotionClassifier(num_classes=5).to(device)

# Optimizers
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
optimizer_cls = torch.optim.Adam(emotion_classifier.parameters(), lr=1e-3)

# Train 1 epoch
train(model, emotion_classifier, dataloader, optimizer, optimizer_cls, device, num_epochs=1)
