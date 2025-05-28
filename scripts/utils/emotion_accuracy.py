import torch
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
