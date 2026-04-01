import flax.linen as nn
import jax
import jax.numpy as jnp
from typing import Sequence

class Linear(nn.Module):
    dout: int

    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.dout)(x)

class MLP(nn.Module):
    hidden_dims: Sequence[int]  # ex: [64, 32, 10]
    dout: int                  # final output

    @nn.compact
    def __call__(self, x):
        for h in self.hidden_dims:
            x = Linear(h)(x)
            x = nn.relu(x)
        x = Linear(self.dout)(x)
        return x