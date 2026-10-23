import jax
import jax.numpy as jnp
import numpy as np

from optim_env.interpretability.activation_maximization import (
    layer_sizes, maximize_activation, unit_activation)
from optim_env.jax_env.models.cnn import LeNet5


def test_maximize_activation_raises_each_unit_and_keeps_pixel_range():
    model = LeNet5(dout=10)
    params = model.init(jax.random.PRNGKey(0), jnp.zeros((1, 784)))
    assert layer_sizes(params) == {"Conv_0": 6, "Conv_1": 16, "Dense_0": 120, "Dense_1": 84, "Dense_2": 10}

    units = [0, 3, 5]
    images, reached = maximize_activation(model, params, "Conv_1", units, steps=64)
    start = unit_activation(model, params, jnp.full((3, 784), jax.nn.sigmoid(-2.0)), "Conv_1")[jnp.arange(3), jnp.array(units)]

    assert images.shape == (3, 28, 28)
    assert float(images.min()) >= 0 and float(images.max()) <= 1
    assert np.all(np.asarray(reached) > np.asarray(start))


def test_saliency_of_a_linear_model_is_its_weight_column():
    from optim_env.interpretability.saliency import saliency
    from optim_env.jax_env.models.linear import Linear

    model = Linear(dout=3)
    params = model.init(jax.random.PRNGKey(0), jnp.zeros((1, 5)))
    X = jax.random.normal(jax.random.PRNGKey(1), (4, 5))
    kernel = params["params"]["Dense_0"]["kernel"]
    np.testing.assert_allclose(saliency(model, params, X, [0, 1, 2, 1]),
                               np.abs(kernel[:, [0, 1, 2, 1]].T), rtol=1e-5)



def test_on_object_share_counts_saliency_on_bright_pixels():
    from optim_env.tools.interpretability_grid import on_object_share

    X = np.array([[0.0, 0.5, 0.9, 0.05]])
    assert on_object_share(np.array([[1.0, 1.0, 2.0, 0.0]]), X) == 0.75
