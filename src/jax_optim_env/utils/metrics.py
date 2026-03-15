import jax.numpy as jnp
import jax

@jax.jit
def accuracy(logits, targets):
    """Calculates accuracy for classification tasks."""
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(preds == targets)

@jax.jit
def mse(preds, targets):
    """Calculates Mean Squared Error for regression tasks."""
    return jnp.mean(jnp.square(preds - targets))

@jax.jit
def mae(preds, targets):
    """Calculates Mean Absolute Error for regression tasks."""
    return jnp.mean(jnp.abs(preds - targets))
