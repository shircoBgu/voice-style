import argparse
import os
from scripts.train import train_model
from scripts.inference import run_inference

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
    if args.mode == 'train':
        train_model(args.config)
    elif args.mode == 'inference':
        run_inference(args.config)

if __name__ == "__main__":
    main()
