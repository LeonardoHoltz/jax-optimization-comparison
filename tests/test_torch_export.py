import jax
import jax.numpy as jnp
import numpy as np
import torch

from jax_optim_env.jax_env.models.linear import Linear, MLP
from jax_optim_env.utils.torch_export import flax_to_torch


def test_flax_to_torch_linear_matches_output():
    model = Linear(dout=3)
    params = model.init(jax.random.PRNGKey(0), jnp.zeros((1, 5)))
    torch_model = flax_to_torch(model, params)

    x = jax.random.normal(jax.random.PRNGKey(1), (4, 5))
    jax_out = np.asarray(model.apply(params, x))
    torch_out = torch_model(torch.from_numpy(np.array(x))).detach().numpy()

    np.testing.assert_allclose(jax_out, torch_out, atol=1e-5)


def test_flax_to_torch_mlp_matches_output():
    model = MLP(hidden_dims=[8, 4], dout=2, dropout=0.3)
    params = model.init(jax.random.PRNGKey(0), jnp.zeros((1, 6)))
    torch_model = flax_to_torch(model, params)

    x = jax.random.normal(jax.random.PRNGKey(2), (5, 6))
    jax_out = np.asarray(model.apply(params, x, deterministic=True))
    torch_out = torch_model(torch.from_numpy(np.array(x))).detach().numpy()

    np.testing.assert_allclose(jax_out, torch_out, atol=1e-5)


def test_flax_to_torch_mlp_no_dropout_layer_when_disabled():
    model = MLP(hidden_dims=[8], dout=2, dropout=0.0)
    params = model.init(jax.random.PRNGKey(0), jnp.zeros((1, 6)))
    torch_model = flax_to_torch(model, params)

    assert not any(isinstance(m, torch.nn.Dropout) for m in torch_model)


def test_flax_to_torch_mlp_is_eval_mode():
    model = MLP(hidden_dims=[8], dout=2, dropout=0.5)
    params = model.init(jax.random.PRNGKey(0), jnp.zeros((1, 6)))
    torch_model = flax_to_torch(model, params)

    assert not torch_model.training
