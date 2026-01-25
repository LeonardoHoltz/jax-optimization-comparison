from flax import nnx
import jax
import jax.numpy as jnp

class Linear(nnx.Module):
    def __init__(self, din: int, dout: int, rngs: nnx.Rngs):
        self.w = nnx.Param(rngs.uniform((din, dout)))
        self.b = nnx.Param(jnp.zeros((dout,)))
        self.din, self.dout = din, dout
    
    def __call__(self, x: jax.Array):
        return x @ self.w + self.b[None]

model = Linear(2, 5, rngs=nnx.Rngs(42))
y = model(x=jnp.ones((1, 2)))

print(y)
nnx.display(model)
