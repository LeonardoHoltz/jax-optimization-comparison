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


def test_saliency_of_a_linear_model_is_its_signed_weight_column():
    from optim_env.interpretability.saliency import saliency
    from optim_env.jax_env.models.linear import Linear

    model = Linear(dout=3)
    params = model.init(jax.random.PRNGKey(0), jnp.zeros((1, 5)))
    X = jax.random.normal(jax.random.PRNGKey(1), (4, 5))
    kernel = params["params"]["Dense_0"]["kernel"]
    np.testing.assert_allclose(saliency(model, params, X, [0, 1, 2, 1]),
                               kernel[:, [0, 1, 2, 1]].T, rtol=1e-5)



def test_on_object_share_counts_saliency_on_bright_pixels():
    from optim_env.tools.interpretability_grid import on_object_share

    X = np.array([[0.0, 0.5, 0.9, 0.05]])
    assert on_object_share(np.array([[1.0, 1.0, 2.0, 0.0]]), X) == 0.75


def test_integrated_gradients_add_up_to_the_score_change():
    from optim_env.interpretability.saliency import integrated_gradients

    model = LeNet5(dout=10)
    params = model.init(jax.random.PRNGKey(0), jnp.zeros((1, 784)))
    X = jax.random.uniform(jax.random.PRNGKey(1), (2, 784))
    ig = integrated_gradients(model, params, X, [3, 7], steps=256)
    logits, base = model.apply(params, X), model.apply(params, jnp.zeros((1, 784)))[0]
    np.testing.assert_allclose(ig.sum(1), [logits[0, 3] - base[3], logits[1, 7] - base[7]], rtol=2e-2, atol=1e-3)


def test_grad_cam_follows_the_channel_the_class_reads():
    from optim_env.interpretability.saliency import grad_cam
    from optim_env.jax_env.models.cnn import CNN

    # Channel 0 copies the pixel, channel 1 is 1 - pixel. Class 1 reads only channel 0 and
    # class 0 only channel 1, so on an image lit in its top-left quarter, class 1's map sits
    # in that quarter and class 0's map everywhere else.
    model = CNN(input_shape=(28, 28, 1), features=(2,), dout=2)
    p = jax.tree_util.tree_map(jnp.zeros_like, model.init(jax.random.PRNGKey(0), jnp.zeros((1, 784))))
    conv = p["params"]["Conv_0"]
    conv["kernel"] = conv["kernel"].at[1, 1, 0, 0].set(1.0).at[1, 1, 0, 1].set(-1.0)
    conv["bias"] = conv["bias"].at[1].set(1.0)
    reads = jnp.zeros((14 * 14, 2, 2)).at[:, 0, 1].set(1.0).at[:, 1, 0].set(1.0)
    p["params"]["Dense_0"]["kernel"] = reads.reshape(-1, 2)
    x = jnp.zeros((28, 28)).at[:14, :14].set(1.0).reshape(1, -1)
    lit, dark = np.asarray(grad_cam(model, p, jnp.concatenate([x, x]), [1, 0])).reshape(2, 28, 28)
    assert lit[:12, :12].min() > 0 and lit[16:].max() == 0 and lit[:, 16:].max() == 0
    assert dark[:12, :12].max() == 0 and dark[16:].min() > 0


def test_same_sign_share_tells_patches_from_checkerboard():
    from optim_env.tools.interpretability_grid import same_sign_share

    X = np.ones((1, 16))
    checker = np.indices((4, 4)).sum(0) % 2 * 2 - 1.0
    assert same_sign_share(checker.reshape(1, -1), X, side=4) == 0.0
    assert same_sign_share(np.ones((1, 16)), X, side=4) == 1.0
    assert same_sign_share(np.r_[np.ones(8), np.zeros(8)][None], X, side=4) == 1.0  # zeros ignored


def test_class_means_and_match():
    from optim_env.tools.interpretability_grid import class_means, match

    X = np.array([[1.0, 0.0], [3.0, 0.0], [0.0, 2.0]])
    means = class_means(X, np.array([0, 0, 1]), 2)
    np.testing.assert_allclose(means, [[1, 0], [0, 1]])
    assert match(np.array([[5.0, 1.0], [0.0, 3.0]]), means) == 1.0
