import torch.nn.functional as F


def l2_loss(pred, y):
    """PyTorch mirror of jax_optim_env.utils.loss.l2_loss. Keeps the same 0.5x factor
    optax.l2_loss uses (mean squared error = 2x this), so JAX/PyTorch training curves
    stay directly comparable."""
    if pred.ndim > y.ndim:
        pred = pred.reshape(y.shape)
    return 0.5 * F.mse_loss(pred, y)


def cross_entropy_loss(logits, y):
    return F.cross_entropy(logits, y.long())


def binary_cross_entropy_loss(logits, y):
    return F.binary_cross_entropy_with_logits(logits, y.float())
