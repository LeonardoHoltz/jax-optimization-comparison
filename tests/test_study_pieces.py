import json

import jax
import jax.numpy as jnp
import numpy as np
import optuna
import pytest
from jax.flatten_util import ravel_pytree

from optim_env.datasets.tabm import TabMDataModule
from optim_env.datasets.text import TextDataModule
from optim_env.jax_env.models.cnn import LeNet5
from optim_env.jax_env.models.text import FastText, KimCNN


def _tabm_files(ds_dir, rng):
    for split, n in (("train", 60), ("val", 20), ("test", 20)):
        num = rng.normal(size=(n, 3)).astype(np.float32)
        num[:, 2] = 7.0  # constant column: TabM drops it
        np.save(ds_dir / f"X_num_{split}.npy", num)
        np.save(ds_dir / f"X_bin_{split}.npy", rng.integers(0, 2, size=(n, 1)).astype(np.float32))
        np.save(ds_dir / f"Y_{split}.npy", (rng.normal(size=n) * 50 + 200).astype(np.float32))


def test_noisy_quantile_transforms_numeric_columns_only(tmp_path):
    (tmp_path / "toy").mkdir()
    _tabm_files(tmp_path / "toy", np.random.default_rng(0))
    m = TabMDataModule(name="toy", num_features=3, num_classes=0, root=tmp_path, val_split=0.0,
                       batch_size=8, seed=0, num_policy="noisy-quantile")

    X = m.train_data["X"]
    assert X.shape == (60, 3)  # 2 numeric (constant one dropped) + 1 binary
    assert set(np.unique(X[:, 2])) <= {0.0, 1.0}  # binary column untouched
    assert abs(X[:, :2].mean()) < 0.3  # numeric mapped to ~N(0, 1)
    # Targets standardized with train statistics only.
    assert m.train_data["y"].mean() == pytest.approx(0.0, abs=1e-5)


def test_glove_mean_ignores_padding_and_words_without_vectors(tmp_path):
    ds = tmp_path / "toy"
    ds.mkdir()
    glove = np.zeros((5, 100), dtype=np.float32)
    glove[2], glove[3] = 1.0, 3.0  # ids 0/1 are padding/unknown, id 4 has no GloVe vector
    np.save(ds / "glove100.npy", glove)
    ids = np.array([[2, 3, 4, 1, 0, 0]], dtype=np.int32)
    for split in ("train", "val", "test"):
        np.save(ds / f"{split}_ids.npy", ids)
        np.save(ds / f"{split}_y.npy", np.array([1], dtype=np.int32))
    (ds / "meta.json").write_text(json.dumps({"vocab_size": 5}))

    m = TextDataModule(name="toy", num_features=100, num_classes=2, root=tmp_path, val_split=0.0,
                       batch_size=1, seed=0, features="glove_mean", vocab_size=5)
    np.testing.assert_allclose(m.train_data["X"][0], 2.0)  # mean of 1.0 and 3.0


@pytest.mark.parametrize("model, x, n", [
    (LeNet5(10), jnp.zeros((2, 784)), 61_706),
    (FastText(vocab_size=100, dout=4), jnp.ones((2, 7)), 100 * 10 + 10 * 4 + 4),
])
def test_paper_models_have_the_published_size(model, x, n):
    params = model.init(jax.random.PRNGKey(0), x)
    assert ravel_pytree(params)[0].size == n
    assert model.apply(params, x).shape == (2, model.dout)


def test_text_models_ignore_padding():
    # Extra padding (id 0) must not change the prediction. The shorter input still has
    # width-1 trailing pads, so every filter sees each word-start window in both.
    for model in (FastText(vocab_size=50, dout=3), KimCNN(vocab_size=50, dout=3, embed_dim=8, features=4)):
        short = jnp.array([[5, 6, 7, 8, 9, 0, 0, 0, 0]])
        padded = jnp.array([[5, 6, 7, 8, 9] + [0] * 20])
        params = model.init(jax.random.PRNGKey(0), short)
        np.testing.assert_allclose(model.apply(params, short), model.apply(params, padded), rtol=1e-5)


def test_hpo_best_params_map_back_to_overrides():
    from optim_env.tools import hpo

    class Args:
        optimizer, model, lr_min, lr_max, population_size_max = "jax/pgpe", "jax/mlp_small", 1e-4, 1e-2, 64

    sgd = hpo.search_space(Args, "SGD")(optuna.trial.FixedTrial({"learning_rate": 1e-3, "weight_decay": 1e-4, "dropout": 0.2}))
    assert sgd == {"learning_rate": 1e-3, "weight_decay": 1e-4, "model.dropout": 0.2}
    ne = hpo.search_space(Args, "NE")(optuna.trial.FixedTrial({"population_size": 32, "init_std_multiplier": 1.0, "es_learning_rate": 0.01}))
    assert ne == {"optimizer.population_size": 32, "init_std_multiplier": 1.0, "es_learning_rate": 0.01}
