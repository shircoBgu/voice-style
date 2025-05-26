# This script is now used only as a callable function from main.py
# Argument parsing is handled in main.py

from inference.converter import VoiceConverter
from models.autovc import AutoVC

def inference(config, source_path, target_path, emotion_label, output_path):
    """
    Perform voice conversion given config and input/output paths.

    Args:
        config (dict): Loaded configuration dictionary.
        source_path (str): Path to source audio file.
        target_path (str): Path to target speaker audio file.
        emotion_label (int): Target emotion label (0-4).
        output_path (str): Path to save the converted audio.
    """
    # Initialize the voice converter
    converter = VoiceConverter(config)
    converter.load_autovc(lambda: AutoVC())  # Load trained AutoVC model
    converter.load_hifigan()  # Load HiFi-GAN vocoder

    # Perform voice conversion
    converter.convert(
        source_path=source_path,
        target_path=target_path,
        emotion_label=emotion_label,
        output_path=output_path
    )
