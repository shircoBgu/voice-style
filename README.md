# Voice Style Transfer with Emotion Conditioning

This project implements a modified version of [AutoVC](https://github.com/auspicious3000/autovc) for voice conversion with additional conditioning on **speaker identity** and **emotional style**. It enables transferring both the voice and emotional tone from one speaker to another while preserving the linguistic content.

The project also integrates a HiFi-GAN vocoder for high-quality waveform synthesis and supports training over multiple emotion-labeled datasets.

## Features

- Based on AutoVC: encoder → bottleneck → decoder architecture
- Conditioning on:
  - Source mel-spectrogram (linguistic content)
  - Target speaker identity (via embeddings or classifier)
  - Target emotion (via Emotion2Vec or emotion classifier)
- Reconstruction loss, content consistency loss, and emotion embedding loss
- HiFi-GAN vocoder for mel-to-wav synthesis
- Training on multiple datasets (e.g., IEMOCAP, CREMA-D, VCTK)
- Supports inference from `.wav` or `.npy` files
- Visualization of spectrogram output

## Directory Structure
