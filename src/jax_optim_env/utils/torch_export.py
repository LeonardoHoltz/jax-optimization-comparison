import numpy as np
import torch
import torch.nn as nn

from jax_optim_env.models.linear import Linear, MLP


def _dense_to_torch_linear(dense_params) -> nn.Linear:
    """Build a torch.nn.Linear with weights copied from a single Flax `nn.Dense`
    submodule's params. Flax's kernel is shaped (in_features, out_features); torch's
    Linear.weight is (out_features, in_features), hence the transpose."""
    kernel = np.asarray(dense_params["kernel"])
    bias = np.asarray(dense_params["bias"])
    in_features, out_features = kernel.shape
    layer = nn.Linear(in_features, out_features)
    with torch.no_grad():
        layer.weight.copy_(torch.from_numpy(kernel.T.copy()))
        layer.bias.copy_(torch.from_numpy(bias.copy()))
    return layer


def flax_to_torch(model, params) -> nn.Module:
    """Convert a `jax_optim_env.models.linear.Linear` or `MLP` Flax model, plus its
    trained params, into an equivalent PyTorch `nn.Sequential` with the same
    architecture and copied weights. The returned module is in `eval()` mode.

    `params` may be either the raw dict returned by `model.init(...)`/`model.apply(...)`
    (with the standard Flax `{"params": {...}}` wrapper) or an already-unwrapped params
    tree -- both are accepted.
    """
    if "params" in params:
        params = params["params"]

    if isinstance(model, MLP):
        n_dense = len(model.hidden_dims) + 1
        linear_param_trees = [params[f"Linear_{i}"] for i in range(n_dense)]
        dropout = model.dropout
    elif isinstance(model, Linear):
        linear_param_trees = [params]
        dropout = 0.0
    else:
        raise TypeError(
            f"flax_to_torch only supports jax_optim_env Linear/MLP models, got {type(model)}"
        )

    # Each `Linear` submodule's params contain exactly one nested Dense submodule --
    # unpacking (rather than indexing by a hardcoded "Dense_0") means this stays correct
    # even if Flax's internal auto-naming ever changes.
    dense_params_list = []
    for linear_params in linear_param_trees:
        (dense_params,) = linear_params.values()
        dense_params_list.append(dense_params)

    modules = []
    n_layers = len(dense_params_list)
    for i, dense_params in enumerate(dense_params_list):
        modules.append(_dense_to_torch_linear(dense_params))
        if i < n_layers - 1:
            # Matches MLP.__call__: ReLU (+ Dropout, if enabled) after every hidden
            # layer, but not after the final output layer.
            modules.append(nn.ReLU())
            if dropout > 0:
                modules.append(nn.Dropout(p=dropout))

    torch_model = nn.Sequential(*modules)
    torch_model.eval()
    return torch_model


def flax_checkpoint_to_torch(model, checkpoint_path, dummy_input) -> nn.Module:
    """Load a Flax params checkpoint saved by `jax_optim_env.utils.checkpointing.save_checkpoint`
    (as used by SGDTrainer/ESTrainer for `best_model.msgpack`) and convert it to an
    equivalent PyTorch module via `flax_to_torch`.

    `dummy_input` is a JAX array with the model's expected input shape (e.g.
    `jnp.zeros((1, num_features))`), used only to build a template pytree for
    deserializing the checkpoint -- its values are discarded.
    """
    import jax

    from jax_optim_env.utils.checkpointing import load_checkpoint

    template = model.init(jax.random.PRNGKey(0), dummy_input)
    params = load_checkpoint(template, checkpoint_path)
    return flax_to_torch(model, params)
