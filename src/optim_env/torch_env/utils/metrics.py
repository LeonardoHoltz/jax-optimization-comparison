import torch


def accuracy(logits, targets):
    """Calculates accuracy for classification tasks."""
    preds = torch.argmax(logits, dim=-1)
    return (preds == targets).float().mean()


def mse(preds, targets):
    """Calculates Mean Squared Error for regression tasks."""
    return torch.mean((preds - targets) ** 2)


def mae(preds, targets):
    """Calculates Mean Absolute Error for regression tasks."""
    return torch.mean(torch.abs(preds - targets))
