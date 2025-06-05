import argparse
import json
import torch
from torch.utils.data import DataLoader
from models.emotion_classifier import EmotionClassifier
from models.autoVC.autovc import AutoVC
from scripts.train import train
from scripts.inference import inference
from scripts.utils.mel_dataset import MelDataset

def parse_args():
    # Argument parser for training and inference modes
    parser = argparse.ArgumentParser(description="Voice Style Transfer with Emotion Conditioning")
    parser.add_argument('--mode', type=str, choices=['train', 'inference'], required=True,
                        help='Run mode: train or inference')
    parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
    parser.add_argument('--source', type=str, help='Path to source audio')
    parser.add_argument('--target', type=str, help='Path to target speaker audio')
    parser.add_argument('--emotion', type=int, help='Emotion label (0-4)')
    parser.add_argument('--output', type=str, help='Path to save converted audio')
    return parser.parse_args()

def main():
    args = parse_args()

    # Load configuration from JSON
    with open(args.config) as f:
        config = json.load(f)

    # Set device to CUDA if available, else fallback to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === TRAINING MODE ===
    if args.mode == 'train':
        # Load dataset path and prepare dataloader
        csv_path = config["dataset"].get("iemocap_path")
        dataset = MelDataset(csv_path)
        dataloader = DataLoader(dataset,
                                batch_size=config["training"]["batch_size"],
                                shuffle=True,
                                drop_last=True)

        # Initialize models
        model = AutoVC().to(device)
        emotion_classifier = EmotionClassifier(num_emotions=config["model"]["num_emotions"]).to(device)

        # Initialize optimizers
        lr = config["training"]["learning_rate"]
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        optimizer_cls = torch.optim.Adam(emotion_classifier.parameters(), lr=lr)

        # Begin training
        train(model, emotion_classifier, dataloader,
              optimizer, optimizer_cls, device,
              num_epochs=config["training"]["epochs"],
              lambda_ce=0.5,
              lambda_spk=0.5,
              checkpoint_dir=config["training"]["checkpoint_dir"])

    # === INFERENCE MODE ===
    elif args.mode == 'inference':
        # Ensure all necessary arguments are provided for inference
        if not (args.source and args.target and args.output and args.emotion is not None):
            raise ValueError("Inference mode requires --source, --target, --emotion, and --output arguments.")

        # Run inference
        inference(config=config,
                  source_path=args.source,
                  target_path=args.target,
                  emotion_label=args.emotion,
                  output_path=args.output)

if __name__ == "__main__":
    main()
