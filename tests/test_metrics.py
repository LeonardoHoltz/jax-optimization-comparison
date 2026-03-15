import jax.numpy as jnp
import numpy as np
from jax_optim_env.utils.metrics import accuracy, mse, mae

def test_accuracy():
    logits = jnp.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0]
    ])
    targets = jnp.array([0, 1, 2, 1]) # last one is wrong
    acc = accuracy(logits, targets)
    assert np.isclose(acc, 0.75)

def test_mse():
    preds = jnp.array([1.0, 2.0, 3.0])
    targets = jnp.array([1.5, 2.5, 3.5])
    val = mse(preds, targets)
    assert np.isclose(val, 0.25)

def test_mae():
    preds = jnp.array([1.0, 2.0, 3.0])
    targets = jnp.array([1.5, 2.5, 3.5])
    val = mae(preds, targets)
    assert np.isclose(val, 0.5)
