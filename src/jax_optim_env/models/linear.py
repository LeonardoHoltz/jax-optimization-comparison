import flax.linen as nn
import jax
import jax.numpy as jnp
from typing import Sequence

class Linear(nn.Module):
    dout: int

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        return nn.Dense(self.dout)(x)

class MLP(nn.Module):
    hidden_dims: Sequence[int]  # ex: [64, 32, 10]
    dout: int                  # final output
    dropout: float = 0.0       # matches the paper's MLPBlock: Dropout(ReLU(Linear(x)))

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        for h in self.hidden_dims:
            x = Linear(h)(x)
            x = nn.relu(x)
            if self.dropout > 0:
                x = nn.Dropout(rate=self.dropout, deterministic=deterministic)(x)
        x = Linear(self.dout)(x)
        return x