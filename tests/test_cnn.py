import jax
import jax.numpy as jnp
import numpy as np
import pytest

from optim_env.datasets.mnist import MNISTDataModule
from optim_env.jax_env.models.cnn import CNN


def test_cnn_reshapes_flat_input_and_outputs_logits():
    # The shared data pipeline hands out 2-D X, so the model owns the image reshape.
    model = CNN(input_shape=(28, 28, 1), features=(16, 32), dout=10)
    params = model.init(jax.random.PRNGKey(0), jnp.zeros((1, 784)))
    out = model.apply(params, jnp.zeros((5, 784)), deterministic=True)
    assert out.shape == (5, 10)


def test_cnn_stays_small_enough_for_evolutionary_search():
    # ES searches the raw parameter vector and degrades sharply with its dimension, so
    # this architecture is deliberately an order of magnitude below the tabular MLP's 235K.
    from jax.flatten_util import ravel_pytree

    model = CNN(input_shape=(28, 28, 1), features=(16, 32), dout=10)
    params = model.init(jax.random.PRNGKey(0), jnp.zeros((1, 784)))
    flat, _ = ravel_pytree(params)
    assert flat.shape[0] < 50_000


def _write_fake_mnist(tmp_path, n_train=40, n_test=12):
    ds_dir = tmp_path / "mnist"
    ds_dir.mkdir()
    rng = np.random.default_rng(0)
    np.savez(
        ds_dir / "mnist.npz",
        x_train=rng.integers(0, 256, (n_train, 28, 28), dtype=np.uint8),
        y_train=rng.integers(0, 10, (n_train,), dtype=np.uint8),
        x_test=rng.integers(0, 256, (n_test, 28, 28), dtype=np.uint8),
        y_test=rng.integers(0, 10, (n_test,), dtype=np.uint8),
    )
    return ds_dir


def test_mnist_keeps_official_test_split_and_carves_val_from_train(tmp_path):
    _write_fake_mnist(tmp_path, n_train=40, n_test=12)
    ds = MNISTDataModule(
        name="mnist", num_features=784, num_classes=10, root=tmp_path,
        val_split=0.0, batch_size=4, seed=0, val_size=10,
    )

    # Official test set is preserved untouched; val comes out of train.
    assert ds.test_data["X"].shape == (12, 784)
    assert ds.val_data["X"].shape == (10, 784)
    assert ds.train_data["X"].shape == (30, 784)


def test_mnist_scales_pixels_to_unit_range(tmp_path):
    _write_fake_mnist(tmp_path)
    ds = MNISTDataModule(
        name="mnist", num_features=784, num_classes=10, root=tmp_path,
        val_split=0.0, batch_size=4, seed=0, val_size=10,
    )
    X = ds.train_data["X"]
    assert X.dtype == np.float32
    assert 0.0 <= X.min() and X.max() <= 1.0
    assert ds.train_data["y"].dtype == np.int32


def test_mnist_raises_with_download_hint_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="curl"):
        MNISTDataModule(
            name="mnist", num_features=784, num_classes=10, root=tmp_path,
            val_split=0.0, batch_size=4, seed=0,
        )
