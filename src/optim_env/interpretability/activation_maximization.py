"""Activation maximization: find the input image that drives one unit as high as it goes.

Start from faint noise and run gradient ascent on the pixels, with the weights frozen
(Erhan et al. 2009; Simonyan et al. 2013 for class logits). Three tricks keep the result
readable instead of adversarial noise (Yosinski et al. 2015, Olah et al. 2017):
  - pixels are sigmoid(z), so they stay in the [0, 1] range the models trained on;
  - a total-variation penalty favours smooth strokes over speckle;
  - a penalty on mean brightness keeps only the strokes the unit needs;
  - each step shifts the image by a few random pixels (jitter), so a pattern must work
    at nearby positions, not exploit one exact pixel layout.
Units differ in scale (logits reach ~40, conv channels ~1), so both penalties are
multiplied by the unit's spread over random images. One set of weights then fits all.

A unit is a channel of a Conv layer (averaged over positions) or one output of a Dense
layer. The values are taken before the ReLU; on the last layer they are the class logits.
"""

import math

import jax
import jax.numpy as jnp
import optax


def layer_sizes(params) -> dict[str, int]:
    """{layer name: unit count} for the model's top-level Conv and Dense layers."""
    return {name: p["kernel"].shape[-1] for name, p in params["params"].items() if "kernel" in p}


def unit_activation(model, params, x, layer: str):
    """(batch, units) pre-ReLU values of `layer`; Conv maps are averaged over positions."""
    _, state = model.apply(params, x, deterministic=True, capture_intermediates=lambda m, _: m.name == layer)
    out = state["intermediates"][layer]["__call__"][0]
    return out.reshape(out.shape[0], -1, out.shape[-1]).mean(1)


def total_variation(images):
    """Mean absolute difference between neighbouring pixels, per image."""
    return jnp.abs(jnp.diff(images, axis=1)).mean((1, 2)) + jnp.abs(jnp.diff(images, axis=2)).mean((1, 2))


def maximize_activation(model, params, layer: str, units, image_shape=(28, 28), steps: int = 512,
                        lr: float = 0.05, tv_weight: float = 10.0, brightness_weight: float = 20.0, jitter: int = 1, seed: int = 0):
    """One image per entry of `units`, optimized together. Returns (images in [0, 1] of
    shape (len(units), *image_shape), the activation each image reaches)."""
    units = jnp.asarray(units)
    rows = jnp.arange(len(units))
    key, noise_key, scale_key = jax.random.split(jax.random.PRNGKey(seed), 3)
    random_images = jax.random.uniform(scale_key, (256, math.prod(image_shape)))
    scale = unit_activation(model, params, random_images, layer).std(0)[units]
    # sigmoid(-2) = 0.12: a dark, near-blank start, like the digits' background.
    z = -2.0 + 0.1 * jax.random.normal(noise_key, (len(units), *image_shape))
    tx = optax.adam(lr)

    def objective(z, shift):
        images = jax.nn.sigmoid(z)
        shifted = jnp.roll(images, shift, axis=(1, 2))
        act = unit_activation(model, params, shifted.reshape(len(units), -1), layer)[rows, units]
        penalty = tv_weight * total_variation(images) + brightness_weight * images.mean((1, 2))
        return -(act - scale * penalty).sum()

    @jax.jit
    def step(carry, key):
        z, opt_state = carry
        shift = jax.random.randint(key, (2,), -jitter, jitter + 1)
        grads = jax.grad(objective)(z, shift)
        updates, opt_state = tx.update(grads, opt_state)
        return (optax.apply_updates(z, updates), opt_state), None

    (z, _), _ = jax.lax.scan(step, (z, tx.init(z)), jax.random.split(key, steps))
    images = jax.nn.sigmoid(z)
    return images, unit_activation(model, params, images.reshape(len(units), -1), layer)[rows, units]
