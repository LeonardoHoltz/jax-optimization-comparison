import json
import logging

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

from optim_env.datasets import XYDataset
from optim_env.jax_env.models.linear import Linear
from optim_env.jax_env.trainers.es_trainer import ESTrainer
from optim_env.utils.checkpointing import load_checkpoint


def make_trainer(algorithm, tmp_path, generations=5):
    """ESTrainer without its Hydra-bound __init__: tiny regression data, a Linear model."""
    cfg = OmegaConf.create({
        "seed": 0,
        "generations": generations,
        "eval_every_gen": 1,
        "save_path": str(tmp_path),
        "init_std_multiplier": 1.0,
        "es_learning_rate": 1e-3,
        "dataset": {"num_classes": 0},
        "model": {"_target_": "optim_env.jax_env.models.linear.Linear", "dout": 1},
        "optimizer": {"_target_": f"evosax.algorithms.{algorithm}", "population_size": 8},
        "loss": {"_target_": "optim_env.utils.loss.l2_loss", "_call_": False},
    })
    rng = np.random.default_rng(0)
    X = rng.normal(size=(32, 3)).astype(np.float32)
    y = X @ np.array([1.0, -2.0, 0.5], dtype=np.float32)

    trainer = ESTrainer.__new__(ESTrainer)
    trainer.cfg = cfg
    trainer.logger = logging.getLogger("test")
    trainer.writer = SummaryWriter(str(tmp_path))
    trainer.key = jax.random.PRNGKey(0)
    trainer.train_ds = XYDataset(X, y)
    trainer.train_X, trainer.train_y = jnp.asarray(X[:24]), jnp.asarray(y[:24])
    trainer.val_X, trainer.val_y = jnp.asarray(X[24:]), jnp.asarray(y[24:])
    trainer.prepare_model()
    trainer.prepare_loss_fn()
    trainer._compile()
    trainer.best_fitness = float("inf")
    trainer.best_params = None
    return trainer


def test_population_based_std_decays_but_distribution_based_stays_constant(tmp_path):
    # Open_ES's std divides its gradient estimate twice; decaying it blew val loss up to
    # millions. SimpleGA's std is the raw mutation size and must shrink to stop drift.
    ga = make_trainer("SimpleGA", tmp_path / "ga")
    es = make_trainer("Open_ES", tmp_path / "es")

    ga_start, ga_late = float(ga.strategy.std_schedule(0)), float(ga.strategy.std_schedule(100))
    es_start, es_late = float(es.strategy.std_schedule(0)), float(es.strategy.std_schedule(100))

    assert ga_late == pytest.approx(0.01 * ga_start, rel=1e-3)
    assert es_late == pytest.approx(es_start)
    assert 0 < es_start < 1.0  # model-matched scale, not evosax's generic 1.0


def test_pso_starts_with_a_diverse_population(tmp_path):
    # Identical particles make every PSO velocity term zero forever.
    pso = make_trainer("PSO", tmp_path)
    assert float(pso.state.population.std(axis=0).mean()) > 0


@pytest.mark.parametrize("algorithm", ["Sep_CMA_ES", "SimpleGA"])
def test_saved_checkpoint_is_the_one_validation_picked(tmp_path, algorithm):
    trainer = make_trainer(algorithm, tmp_path)
    trainer.fit()

    metrics = json.loads((tmp_path / "metrics.json").read_text())
    template = Linear(dout=1).init(jax.random.PRNGKey(0), jnp.zeros((1, 3)))
    saved = load_checkpoint(template, tmp_path / "best_model.msgpack")
    pred = Linear(dout=1).apply(saved, trainer.val_X)
    saved_val_loss = float(trainer.loss_fn(pred, trainer.val_y))

    assert saved_val_loss == pytest.approx(metrics["best_val_loss"], rel=1e-5)
