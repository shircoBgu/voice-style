import torch
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import torchaudio

# Initialize the pipeline globally
emotion2vec_pipeline = pipeline(
    task=Tasks.emotion_recognition,
    model="iic/emotion2vec_base"
)


def resample_to_16k(wav_tensor, orig_sr=22050, target_sr=16000):
    """
    Resamples waveform tensor from orig_sr to target_sr.
    """
    if not isinstance(wav_tensor, torch.Tensor):
        wav_tensor = torch.tensor(wav_tensor)
    resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)
    return resampler(wav_tensor).float()


def extract_emotion_embedding(input_data, sr=16000, pipe=None):
    """
    Extracts a 64-dimensional emotion embedding from either:
    - a .wav file path
    - a waveform (numpy or torch tensor)

    Args:
        input_data: str (wav path) or waveform (np.ndarray or torch.Tensor)
        sr (int): sample rate of waveform. Should be 16000 for emotion2vec.
        pipe: optional pipeline instance

    Returns:
        torch.Tensor: Emotion embedding of shape (64,)
    """
    if pipe is None:
        pipe = emotion2vec_pipeline

    # Load or use waveform
    if isinstance(input_data, str):  # assume it's a path
        result = pipe(
            input_data,
            extract_embedding=True,
            granularity="utterance",
            output_dir="./outputs"
        )
    else:
        # waveform input: convert to numpy
        if isinstance(input_data, torch.Tensor):
            input_data = input_data.detach().cpu().numpy()
        if sr != 16000:
            input_data = resample_to_16k(input_data, orig_sr=sr).numpy()
        result = pipe(
            input_data,
            extract_embedding=True,
            granularity="utterance",
            output_dir="./outputs"
        )

    return torch.tensor(result['embedding']).float()


# Accuracy test
def emotion_accuracy(logits, emotion_labels):
    """
    Compute accuracy given model output logits and true labels.

    Args:
        logits: Tensor of shape [B, 9] – raw model outputs
        emotion_labels: Tensor of shape [B] – true class indices (0 to C-1)

    Returns:
        accuracy: Float – percentage of correct predictions
    """
    preds = torch.argmax(logits, dim=1)  # shape: [B]
    correct = (preds == emotion_labels).sum().item()  # number of correct predictions
    total = emotion_labels.size(0)  # batch size
    accuracy = correct / total
    return accuracy
