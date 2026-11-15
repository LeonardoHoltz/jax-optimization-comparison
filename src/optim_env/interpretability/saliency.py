"""Attribution maps: which input pixels a trained model's class score depends on.

saliency: d score / d pixel for one image and one class logit (Simonyan et al. 2014),
kept signed: positive where brightening the pixel raises the score, negative where it
lowers it. Quick to compute; noisy, and blind to pixels whose effect has saturated.

integrated_gradients: the gradient averaged along the straight path from a black image
to the input, times the input (Sundararajan et al. 2017). Signed, and it adds up to
score(input) - score(black), so saturation can't hide a pixel. A pixel that is black in
the input gets exactly zero.

grad_cam: the last conv layer's feature maps, weighted by how much each channel's
gradient pushes the score, summed and clipped at zero (Selvaraju et al. 2017). Coarse
(the conv grid, upsampled), but it shows the regions the model uses rather than single
pixels.
"""

import flax.linen as nn
import jax
import jax.numpy as jnp


def saliency(model, params, X, targets):
    """(n, features) signed gradient of each row's target-class logit."""
    grad = jax.grad(lambda x, t: model.apply(params, x[None], deterministic=True)[0, t])
    return jax.jit(jax.vmap(grad))(X, jnp.asarray(targets))


def integrated_gradients(model, params, X, targets, steps=32):
    """(n, features) integrated gradients from an all-zero baseline, midpoint rule."""
    alphas = (jnp.arange(steps) + 0.5) / steps
    grad = jax.grad(lambda x, t: model.apply(params, x[None], deterministic=True)[0, t])

    def one(x, t):
        return x * jax.vmap(lambda a: grad(a * x, t))(alphas).mean(0)

    # 64 images at a time: all at once would hold steps x n activations.
    return jax.jit(lambda X, t: jax.lax.map(lambda a: one(*a), (X, t), batch_size=64))(X, jnp.asarray(targets))


def grad_cam(model, params, X, targets, side=28):
    """(n, side*side) Grad-CAM of the last conv layer, upsampled to the input grid."""
    layer = sorted(k for k in params["params"] if k.startswith("Conv"))[-1]
    capture = lambda mdl, _: mdl.name == layer  # noqa: E731

    def score(delta, x, t):
        # Adds delta to the layer's output, so d score / d delta is the gradient there.
        def add(next_fun, args, kwargs, context):
            out = next_fun(*args, **kwargs)
            return out + delta if context.module.name == layer and context.method_name == "__call__" else out

        with nn.intercept_methods(add):
            logits, state = model.apply(params, x[None], deterministic=True,
                                        capture_intermediates=capture, mutable=["intermediates"])
        return logits[0, t], state["intermediates"][layer]["__call__"][0][0]

    _, state = model.apply(params, X[:1], capture_intermediates=capture, mutable=["intermediates"])
    shape = state["intermediates"][layer]["__call__"][0].shape  # (1, h, w, channels)
    # Each conv block pools by 2, so this conv reads a (side >> index) grid; a VALID conv's
    # smaller output sits centred in it.
    grid = side >> int(layer.split("_")[1])
    pad = (grid - shape[1]) // 2

    def one(x, t):
        g, acts = jax.grad(score, has_aux=True)(jnp.zeros(shape), x, t)
        cam = jax.nn.relu((acts * g[0].mean((0, 1))).sum(-1))
        cam = jnp.pad(cam, ((pad, grid - shape[1] - pad), (pad, grid - shape[2] - pad)))
        return jax.image.resize(cam, (side, side), "bilinear").reshape(-1)

    return jax.jit(jax.vmap(one))(X, jnp.asarray(targets))
