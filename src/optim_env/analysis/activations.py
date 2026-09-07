"""Per-layer hidden activations from a trained model, for representation analysis.

Returns post-activation (post-ReLU) values, which is what "this unit is dead" or "these
two units respond alike" actually refer to. The output layer is excluded -- it has no
activation function and its width is the task's, not a learned representation.

JAX side: Flax's capture_intermediates records each submodule's __call__ output, which for
this MLP is the Dense output *before* ReLU (ReLU is applied in MLP.__call__, outside the
submodule). Post-ReLU is also recoverable from the Dropout submodule's output, but only
when dropout > 0 -- so ReLU is applied here instead, which works either way.
Torch side: forward hooks on the nn.ReLU modules give post-activation values directly.
"""

import numpy as np

from optim_env.analysis.loading import LoadedRun


def hidden_activations(run: LoadedRun, X) -> dict[str, np.ndarray]:
    """{layer_name: (n_samples, n_units)} post-ReLU activations for each hidden layer."""
    if run.framework == "torch":
        return _torch_activations(run, X)
    return _jax_activations(run, X)


def _jax_activations(run: LoadedRun, X) -> dict[str, np.ndarray]:
    import jax
    import jax.numpy as jnp

    _, state = run.model.apply(
        run.params, jnp.asarray(X), deterministic=True, capture_intermediates=True
    )
    intermediates = state["intermediates"]

    n_hidden = len(run.cfg.model.hidden_dims)
    out = {}
    for i in range(n_hidden):  # Linear_{n_hidden} is the output layer -- excluded
        pre_activation = intermediates[f"Linear_{i}"]["__call__"][0]
        out[f"hidden_{i}"] = np.asarray(jax.nn.relu(pre_activation))
    return out


def _torch_activations(run: LoadedRun, X) -> dict[str, np.ndarray]:
    import torch
    import torch.nn as nn

    captured: list[np.ndarray] = []
    handles = [
        module.register_forward_hook(lambda _m, _i, o: captured.append(o.detach().cpu().numpy()))
        for module in run.model.modules()
        if isinstance(module, nn.ReLU)
    ]
    try:
        run.model.eval()
        with torch.no_grad():
            run.model(torch.from_numpy(np.asarray(X, dtype=np.float32)))
    finally:
        for h in handles:
            h.remove()

    return {f"hidden_{i}": act for i, act in enumerate(captured)}
