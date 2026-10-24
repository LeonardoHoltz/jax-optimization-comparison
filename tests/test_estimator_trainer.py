import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import traverse_util
from jax.flatten_util import ravel_pytree

from optim_env.jax_env.trainers.estimator_trainer import DustTrainer, EggrollTrainer
from trainer_helpers import build_trainer

TRAINERS = {
    "EGGROLL": (EggrollTrainer, {"population_size": 64, "rank": 1, "momentum": None}),
    "DUST": (DustTrainer, {"population_size": 64, "sigma": 0.2, "momentum": 0.9}),
}


def make_trainer(algorithm, tmp_path, lr=1e-2, generations=5, **optimizer):
    """Tiny regression data, a small MLP."""
    cls, defaults = TRAINERS[algorithm]
    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, 3)).astype(np.float32)
    y = np.tanh(X @ np.array([1.0, -2.0, 0.5], dtype=np.float32))
    return build_trainer(
        cls, tmp_path, X, y, n_train=48, generations=generations, es_learning_rate=lr,
        model={"_target_": "optim_env.jax_env.models.linear.MLP", "hidden_dims": [16], "dout": 1},
        optimizer={**defaults, **optimizer},
    )


def true_grad(trainer):
    def loss(solution):
        pred = trainer.model.apply(trainer.unravel_fn(solution), trainer.train_X)
        return trainer.loss_fn(pred, trainer.train_y)
    return jax.grad(loss)(trainer.state[0])


def cosine(a, b):
    return float(a @ b / (jnp.linalg.norm(a) * jnp.linalg.norm(b)))


def test_eggroll_low_rank_forward_matches_materialized_weights(tmp_path):
    trainer = make_trainer("EGGROLL", tmp_path, rank=2)
    params = trainer.unravel_fn(trainer.state[0])
    keys = iter(jax.random.split(jax.random.PRNGKey(1), 8))
    eps, perturbed = {}, {}
    for k, v in traverse_util.flatten_dict(params).items():
        if k[-1] == "kernel":
            A, B = jax.random.normal(next(keys), (v.shape[0], 2)), jax.random.normal(next(keys), (v.shape[1], 2))
            eps[k], perturbed[k] = (A, B), v + 0.3 * A @ B.T / jnp.sqrt(2)
        else:
            eps[k] = jax.random.normal(next(keys), v.shape)
            perturbed[k] = v + 0.3 * eps[k]

    # Full fp32 matmuls: the GPU's default TF32 differs between the two orderings by ~1e-4.
    with jax.default_matmul_precision("highest"):
        low_rank = trainer._member_loss(params, eps, 0.3, trainer.train_X, trainer.train_y)
        pred = trainer.model.apply(traverse_util.unflatten_dict(perturbed), trainer.train_X)
    assert float(low_rank) == pytest.approx(float(trainer.loss_fn(pred, trainer.train_y)), rel=1e-5)


# EGGROLL's rank shaping caps it near 0.91 here (5 seeds: 0.90-0.92); a sign or scale
# bug lands near 0 or below.
@pytest.mark.parametrize("algorithm, optimizer, min_cosine", [
    ("EGGROLL", {"population_size": 8192}, 0.8),
    ("DUST", {"population_size": 4096, "sigma": 0.05}, 0.9),
])
def test_gradient_estimate_points_along_the_true_gradient(tmp_path, algorithm, optimizer, min_cosine):
    trainer = make_trainer(algorithm, tmp_path, **optimizer)
    params = trainer.unravel_fn(trainer.state[0])
    grad, _ = trainer._estimate(params, jax.random.PRNGKey(2), trainer.train_X, trainer.train_y)
    assert cosine(ravel_pytree(grad)[0], true_grad(trainer)) > min_cosine


@pytest.mark.parametrize("algorithm", ["EGGROLL", "DUST"])
def test_training_lowers_val_loss_and_saves_the_picked_checkpoint(tmp_path, algorithm):
    trainer = make_trainer(algorithm, tmp_path, lr=0.05, generations=100)
    start_val_loss, _ = trainer.evaluate(trainer.state[0])
    trainer.fit()

    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["best_val_loss"] < 0.8 * start_val_loss
    assert (tmp_path / "best_model.msgpack").exists()


# Conv and Embed go through the same local rule as Dense, so their estimates must also
# point along the true gradient. Inputs are ids for fastText, pixels for the CNNs.
# LeNet perturbs ~6,500 outputs per example, so 4096 draws give a noisier estimate (0.82;
# 0.93 at 16K draws, 0.98 at 65K). A sign or scale bug lands near 0.
@pytest.mark.parametrize("model, n_features, ids, min_cosine", [
    ({"_target_": "optim_env.jax_env.models.cnn.CNN", "input_shape": [6, 6, 1], "features": [3], "dout": 1}, 36, False, 0.9),
    ({"_target_": "optim_env.jax_env.models.cnn.LeNet5", "dout": 1}, 784, False, 0.7),
    ({"_target_": "optim_env.jax_env.models.text.FastText", "vocab_size": 7, "dout": 1, "embed_dim": 4}, 5, True, 0.9),
])
def test_dust_estimate_covers_conv_and_embedding_layers(tmp_path, model, n_features, ids, min_cosine):
    rng = np.random.default_rng(0)
    # 16 training rows: 4096 draws of LeNet's first conv output already take ~1 GB.
    X = (rng.integers(0, 7, size=(24, n_features)) if ids else rng.uniform(size=(24, n_features))).astype(np.float32)
    y = np.tanh(X.mean(1) - X.mean()).astype(np.float32)[:, None]
    trainer = build_trainer(DustTrainer, tmp_path, X, y, n_train=16, model=model,
                            optimizer={"population_size": 4096, "sigma": 0.05, "momentum": 0.9})
    grad, _ = trainer._estimate(trainer.unravel_fn(trainer.state[0]), jax.random.PRNGKey(2), trainer.train_X, trainer.train_y)
    assert cosine(ravel_pytree(grad)[0], true_grad(trainer)) > min_cosine
