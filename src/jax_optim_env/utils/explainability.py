import jax
import jax.numpy as jnp
from typing import Any, Callable

def compute_saliency_map(
    model_apply_fn: Callable,
    params: Any,
    x: jax.Array,
    target_class: int = None
) -> jax.Array:
    """
    Computes the saliency map for a given input x.
    Saliency is defined as the gradient of the target class logit with respect to the input x.
    
    Args:
        model_apply_fn: The model's apply function (e.g., model.apply).
        params: The parameters of the model.
        x: The input features (single instance, not batched).
        target_class: The class index to compute saliency for. If None, uses the predicted class.
        
    Returns:
        The saliency map (gradient) with the same shape as x.
    """
    def get_logit(input_x):
        logits = model_apply_fn(params, input_x)
        if target_class is not None:
            return logits[target_class]
        # Use the class with the highest logit if no target_class is specified
        return jnp.max(logits)

    # Compute gradient with respect to x
    grad_fn = jax.grad(get_logit)
    saliency = grad_fn(x)
    
    return saliency

def compute_batch_saliency(
    model_apply_fn: Callable,
    params: Any,
    X: jax.Array,
    target_classes: jax.Array = None
) -> jax.Array:
    """
    Computes saliency maps for a batch of inputs X.
    """
    if target_classes is None:
        return jax.vmap(lambda x: compute_saliency_map(model_apply_fn, params, x))(X)
    
    return jax.vmap(lambda x, t: compute_saliency_map(model_apply_fn, params, x, t))(X, target_classes)

def compute_integrated_gradients(
    model_apply_fn: Callable,
    params: Any,
    x: jax.Array,
    baseline: jax.Array = None,
    target_class: int = None,
    steps: int = 50
) -> jax.Array:
    """
    Computes Integrated Gradients for a given input x and baseline.
    IG = (x - baseline) * integral(grad(model(baseline + alpha * (x - baseline))))
    """
    if baseline is None:
        baseline = jnp.zeros_like(x)
        
    diff = x - baseline
    alphas = jnp.linspace(0.0, 1.0, steps + 1)
    
    def get_logit(input_x):
        logits = model_apply_fn(params, input_x)
        if target_class is not None:
             return logits[target_class]
        return jnp.max(logits)
        
    grad_fn = jax.grad(get_logit)
    interpolated_inputs = baseline[None, :] + alphas[:, None] * diff[None, :]
    grads = jax.vmap(grad_fn)(interpolated_inputs)
    
    avg_grads = (grads[1:] + grads[:-1]) / 2.0
    avg_grad = jnp.mean(avg_grads, axis=0)
    
    return diff * avg_grad

def compute_batch_ig(
    model_apply_fn: Callable,
    params: Any,
    X: jax.Array,
    baselines: jax.Array = None,
    target_classes: jax.Array = None,
    steps: int = 50
) -> jax.Array:
    """
    Computes Integrated Gradients for a batch of inputs X.
    """
    if baselines is None:
        baselines = jnp.zeros_like(X)
    
    if target_classes is None:
        return jax.vmap(lambda x, b: compute_integrated_gradients(
            model_apply_fn, params, x, baseline=b, steps=steps
        ))(X, baselines)
    
    return jax.vmap(lambda x, b, t: compute_integrated_gradients(
        model_apply_fn, params, x, baseline=b, target_class=t, steps=steps
    ))(X, baselines, target_classes)
