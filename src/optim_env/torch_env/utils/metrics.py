import torch


def accuracy(logits, targets):
    """Calculates accuracy for classification tasks."""
    preds = torch.argmax(logits, dim=-1)
    return (preds == targets).float().mean()

