import logging

import jax
import jax.numpy as jnp
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

from optim_env.datasets import XYDataset


def build_trainer(cls, tmp_path, X, y, n_train, **cfg):
    """An ESTrainer (or subclass) without its Hydra-bound __init__: regression on X, y,
    the first n_train rows for training and the rest for validation. cfg entries
    override the defaults below."""
    trainer = cls.__new__(cls)
    trainer.cfg = OmegaConf.create({
        "seed": 0,
        "generations": 5,
        "eval_every_gen": 1,
        "save_path": str(tmp_path),
        "init_std_multiplier": 1.0,
        "es_learning_rate": 1e-3,
        "dataset": {"num_classes": 0},
        "loss": {"_target_": "optim_env.utils.loss.l2_loss", "_call_": False},
        **cfg,
    })
    trainer.logger = logging.getLogger("test")
    trainer.writer = SummaryWriter(str(tmp_path))
    trainer.key = jax.random.PRNGKey(0)
    trainer.train_ds = XYDataset(X, y)
    trainer.train_X, trainer.train_y = jnp.asarray(X[:n_train]), jnp.asarray(y[:n_train])
    trainer.val_X, trainer.val_y = jnp.asarray(X[n_train:]), jnp.asarray(y[n_train:])
    trainer.prepare_model()
    trainer.prepare_loss_fn()
    trainer._compile()
    trainer.best_fitness = float("inf")
    trainer.best_params = None
    return trainer
