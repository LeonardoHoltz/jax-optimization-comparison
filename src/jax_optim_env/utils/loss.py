import jax.numpy as jnp
import optax

def l2_loss(pred, y):
    loss = optax.l2_loss(pred, y)
    return jnp.mean(loss)

def cross_entropy_loss(logits, y):
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, y)
    return jnp.mean(loss)

def binary_cross_entropy_loss(logits, y):
    loss = optax.sigmoid_binary_cross_entropy(logits, y)
    return jnp.mean(loss)