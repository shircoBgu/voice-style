# emotion_utils.py
import torch
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# Initialize the emotion2vec pipeline once (global singleton)
emotion2vec_pipeline = pipeline(
    task=Tasks.emotion_recognition,
    model="iic/emotion2vec_base"
)


def extract_emotion_embedding(wav_path, pipe=None):
    """
    Extracts a 64-dimensional emotion embedding from a given .wav file.

    Args:
        wav_path (str): Path to the audio file.
        pipe (optional): Provide a pipeline instance if already created.

    Returns:
        torch.Tensor: Emotion embedding of shape (64,)
    """
    if pipe is None:
        pipe = emotion2vec_pipeline

    result = pipe(wav_path, extract_embedding=True)
    emb = result['embedding']  # (64,)
    return torch.tensor(emb).float()


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
