import jax
import jax.numpy as jnp
import numpy as np
import torch

from optim_env.jax_env.models.cnn import CNN, LeNet5
from optim_env.jax_env.models.linear import Linear, MLP
from optim_env.jax_env.models.text import FastText, KimCNN
from optim_env.utils.torch_export import flax_to_torch


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


def _assert_same_output(model, params, x):
    torch_model = flax_to_torch(model, params)
    with jax.default_matmul_precision("highest"):  # GPU convs default to TF32
        jax_out = np.asarray(model.apply(params, x, deterministic=True))
    torch_out = torch_model(torch.from_numpy(np.array(x))).detach().numpy()
    np.testing.assert_allclose(jax_out, torch_out, atol=1e-5)


def test_flax_to_torch_cnn_matches_output():
    model = CNN(input_shape=(28, 28, 1), features=(16, 32), dout=10, dropout=0.2)
    x = jax.random.uniform(jax.random.PRNGKey(1), (3, 784))
    _assert_same_output(model, model.init(jax.random.PRNGKey(0), x), x)


def test_flax_to_torch_lenet5_matches_output():
    model = LeNet5(dout=10)
    x = jax.random.uniform(jax.random.PRNGKey(1), (3, 784))
    _assert_same_output(model, model.init(jax.random.PRNGKey(0), x), x)


# Token rows with trailing padding (id 0) of different lengths, so the masks matter.
TOKENS = jnp.array([[5, 3, 9, 2, 7, 0, 0, 0], [1, 4, 0, 0, 0, 0, 0, 0], [8, 8, 6, 3, 2, 1, 4, 9]], dtype=jnp.float32)


def test_flax_to_torch_fasttext_matches_output():
    model = FastText(vocab_size=10, dout=4)
    _assert_same_output(model, model.init(jax.random.PRNGKey(0), TOKENS), TOKENS)


def test_flax_to_torch_kim_cnn_matches_output():
    model = KimCNN(vocab_size=10, dout=2, embed_dim=12, features=5)
    _assert_same_output(model, model.init(jax.random.PRNGKey(0), TOKENS), TOKENS)
