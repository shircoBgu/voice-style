import argparse
import os
import torch
from scripts.train import train
from models.autovc import AutoVC
from scripts.inference import inference

def parse_args():
    parser = argparse.ArgumentParser(description="Voice Style Transfer with Emotion Conditioning")
    parser.add_argument('--mode', type=str, choices=['train', 'inference'], required=True,
                        help='Run mode: train or inference')
    parser.add_argument('--config', type=str, default='config.json',
                        help='Path to config file')
    return parser.parse_args()

def main():
    # Set up argument parsing (so we can easily switch between training/inference)
    args = parse_args()
    # === Model & Classifier ===
    model = AutoVC().to(device)
    emotion_classifier = EmotionClassifier(num_emotions=config["num_emotions"]).to(device)

    if args.mode == 'train':
        train(model, emotion_classifier, train_loader,
              optimizer, optimizer_cls, device,
              num_epochs=config["num_epochs"],
              lambda_ce=config["lambda_ce"],
              checkpoint_dir=config["checkpoint_dir"])
    elif args.mode == 'inference':
        run_inference(args.config)

if __name__ == "__main__":
    main()
