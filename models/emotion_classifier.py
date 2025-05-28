from torch import nn

class EmotionClassifier(nn.Module):
    def __init__(self, num_classes=9):
    #emotion: angry, excited, fear, sad, surprised, frustrated, happy, disappointed and neutral
        super().__init__()

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # [B, 1, 80, T] → [B, 16, 80, T]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),                # [B, 16, 40, T/2]

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),                # [B, 32, 20, T/4]

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))                # [B, 64, 1, 1]
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):  # x: [B, T, 80]
        x = x.transpose(1, 2).unsqueeze(1)  # → [B, 1, 80, T]
        x = self.cnn_layers(x)
        return self.classifier(x)


  #add to training the loss function
  #model = EmotionClassifier(num_classes=9)
  #dummy_input = torch.randn(16, 100, 80)  # Batch of 16 spectrograms
  #logits = model(dummy_input)               # Output: [8, 9]
  #probs = F.softmax(logits, dim=1)

  #emotion_labels = torch.randint(0, 9, (16,))  # B random labels in [0–8]
  #criterion = nn.CrossEntropyLoss()
  #loss = criterion(logits, emotion_labels)  # where target_labels ∈ [0,8]
