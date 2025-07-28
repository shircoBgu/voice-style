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

```
voice-style-project/
├── config.json
├── main.py
├── requirements.txt
├── README.md
├── data/
├── models/
│   ├── autoVC/
│   │   ├── autovc.py
│   │   ├── content_encoder.py
│   │   ├── decoder.py
│   │   ├── postnet.py
│   │   ├── speaker_encoder.py
│   ├── EmotionClassifier.py
│   └── hifigan_pretrained/
│       ├── config.json
│       └── generator_v1
├── scripts/
│   ├── train.py
│   ├── inference.py
│   └── utils/
│       ├── emotion_accuracy.py
│       ├── mel_dataset.py
│       ├── converter.py
├── notebooks/
│   └── test_autoVC.py
└── tests/

```

---

## Installation

```bash
git clone https://github.com/shircoBgu/voice_style_project.git
cd voice_style_project
pip install -r requirements.txt
```

You must also place a pretrained HiFi-GAN model inside `models/hifigan_pretrained/` with the following files:
- `generator_v1` (model checkpoint)
- `config.json` (vocoder configuration)

---

## Configuration

Edit `config.json` to define training and dataset parameters:

```json
{
  "dataset": {
    "merged_path": "data/union.csv",
    "sample_rate": 16000,
    "n_mels": 80
  },
  "training": {
    "epochs": 100,
    "batch_size": 8,
    "learning_rate": 0.0002,
    "checkpoint_dir": "checkpoints"
  },
  "paths": {
    "hifigan_pretrained": "models/hifigan_pretrained"
  }
}
```

---

## Training

```bash
python main.py --mode train --config config.json --dataset IEMOCAP
```

- Uses content, speaker, and emotion supervision
- Saves loss plots and checkpoints in `checkpoints/`
- Checkpoints include speaker and emotion mappings

---

## Inference

```bash
python main.py --mode inference \
  --config config.json \
  --source data/mels/source.npy \
  --target data/mels/target.npy \
  --output outputs/converted.wav \
  --use_npy
```

- Converts source to target speaker/emotion style
- Saves `converted.wav` and predicted mel spectrogram
- To use `.wav` instead of `.npy`, remove `--use_npy`

---

## Dataset Format

`union.csv` should contain:

```
mel_path,speaker_id,emotion_label
path/to/sample1.npy,Speaker01,happy
path/to/sample2.npy,Speaker02,sad
...
```

All mel-spectrograms must be of shape `(80, T)` and extracted using consistent Librosa/Torchaudio parameters.

---

## Output

During inference, the system:
- Saves the generated waveform to the output path
- Saves the predicted mel-spectrogram as `.png`
- Optionally saves `.npy` of the predicted mel
- Logs min/max/mean/std of the mel output

---

## Checkpoints

To resume training, place existing checkpoints in `checkpoints/`:
- `checkpoint_epochXXX.pt` includes full model state and mappings
- Auto-loaded automatically at start of training or inference

---

## References

- [AutoVC: One-Shot Voice Conversion](https://arxiv.org/abs/1905.03871)
- [HiFi-GAN](https://arxiv.org/abs/2010.05646)
- [Emotion2Vec (ModelScope)](https://modelscope.cn/models/iic/emotion2vec_plus_seed/summary)

---

## Acknowledgments

Thanks to the developers of AutoVC, HiFi-GAN, and Emotion2Vec. This project builds on their foundational work to support expressive and personalized speech synthesis.
