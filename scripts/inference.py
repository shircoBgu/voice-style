# This script is now used only as a callable function from main.py
# Argument parsing is handled in main.py

from scripts.utils.converter import VoiceConverter
from models.autoVC.autovc import AutoVC


def inference(config, source_path, target_path, emotion_label, output_path, use_npy=False):
    """
    Perform voice conversion given config and input/output paths.

    Args:
        config (dict): Loaded configuration dictionary.
        source_path (str): Path to source audio file.
        target_path (str): Path to target speaker audio file.
        emotion_label (str): Target emotion label.
        output_path (str): Path to save the converted audio.
        use_npy (bool): Whether input files are mel spectrograms (npy)
    """
    # Initialize the voice converter

    converter = VoiceConverter(config)
    converter.load_autovc()  # Load trained AutoVC model
    converter.load_hifigan()  # Load HiFi-GAN vocoder
    if isinstance(emotion_label, str):
        try:
            emotion_label = converter.emo2idx[emotion_label]
        except KeyError:
            raise ValueError(f"Invalid emotion label '{emotion_label}'. Available: {list(converter.emo2idx.keys())}")

    # Perform voice conversion
    converter.convert(
        source_path=source_path,
        target_path=target_path,
        emotion_label=emotion_label,
        output_path=output_path,
        use_npy = use_npy
    )
