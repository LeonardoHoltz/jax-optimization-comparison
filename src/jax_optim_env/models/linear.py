import flax.linen as nn
import jax
import jax.numpy as jnp

class Linear(nn.Module):
    dout: int

    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.dout)(x)
