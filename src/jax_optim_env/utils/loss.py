import jax.numpy as jnp

class MSE:
    def __call__(self, pred, y):
        return jnp.mean(jnp.sum((pred - y) ** 2, axis=-1))