import dataclasses

import numpy as np
import torch
import torch.nn as nn

from optim_env.jax_env.models import cnn, text
from optim_env.jax_env.models.linear import Linear, MLP
from optim_env.torch_env.models import cnn as torch_cnn, text as torch_text

# Flax model -> PyTorch twin built with the same arguments and Flax's layer names.
TWINS = {cnn.CNN: torch_cnn.CNN, cnn.LeNet5: torch_cnn.LeNet5,
         text.FastText: torch_text.FastText, text.KimCNN: torch_text.KimCNN}


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


def _copy_by_name(torch_model: nn.Module, params) -> None:
    """Copy each Flax layer into the torch layer of the same name. Kernels go from
    (*spatial, in, out) to torch's (out, in, *spatial)."""
    with torch.no_grad():
        for name, p in params.items():
            layer = getattr(torch_model, name)
            if "embedding" in p:
                layer.weight.copy_(torch.from_numpy(np.asarray(p["embedding"]).copy()))
                continue
            k = np.asarray(p["kernel"])
            layer.weight.copy_(torch.from_numpy(k.transpose(k.ndim - 1, k.ndim - 2, *range(k.ndim - 2)).copy()))
            layer.bias.copy_(torch.from_numpy(np.asarray(p["bias"]).copy()))


def flax_to_torch(model, params) -> nn.Module:
    """Convert any optim_env Flax model (Linear, MLP, CNN, LeNet5, FastText, KimCNN), plus
    its trained params, into a PyTorch module that computes the same function. Linear and
    MLP become an `nn.Sequential`; the rest become their twins from torch_env.models.
    The returned module is in `eval()` mode. For a finished run directory:
    `run = analysis.loading.load_run(d); flax_to_torch(run.model, run.params)`.

    `params` may be either the raw dict returned by `model.init(...)`/`model.apply(...)`
    (with the standard Flax `{"params": {...}}` wrapper) or an already-unwrapped params
    tree -- both are accepted.
    """
    if "params" in params:
        params = params["params"]

    if type(model) in TWINS:
        fields = {f.name: getattr(model, f.name) for f in dataclasses.fields(model) if f.name not in ("parent", "name")}
        torch_model = TWINS[type(model)](**fields)
        _copy_by_name(torch_model, params)
        return torch_model.eval()

    if isinstance(model, MLP):
        n_dense = len(model.hidden_dims) + 1
        linear_param_trees = [params[f"Linear_{i}"] for i in range(n_dense)]
        dropout = model.dropout
    elif isinstance(model, Linear):
        linear_param_trees = [params]
        dropout = 0.0
    else:
        raise TypeError(
            f"flax_to_torch has no PyTorch version of {type(model)}"
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

