"""Saliency maps: which input pixels a trained model's class score depends on.

The map is |d score / d pixel| for one image and one class logit (Simonyan et al.
2014): how far a small change to each pixel would move the score. Quick to compute;
noisy, and blind to pixels whose effect has saturated.
"""

import jax
import jax.numpy as jnp


def saliency(model, params, X, targets):
    """(n, features) |gradient| of each row's target-class logit."""
    grad = jax.grad(lambda x, t: model.apply(params, x[None], deterministic=True)[0, t])
    return jnp.abs(jax.jit(jax.vmap(grad))(X, jnp.asarray(targets)))
